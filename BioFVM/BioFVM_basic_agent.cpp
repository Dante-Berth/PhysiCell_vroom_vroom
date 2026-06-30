/*
#############################################################################
# If you use BioFVM in your project, please cite BioFVM and the version     #
# number, such as below:                                                    #
#                                                                           #
# We solved the diffusion equations using BioFVM (Version 1.1.7) [1]        #
#                                                                           #
# [1] A. Ghaffarizadeh, S.H. Friedman, and P. Macklin, BioFVM: an efficient #
#    parallelized diffusive transport solver for 3-D biological simulations,#
#    Bioinformatics 32(8): 1256-8, 2016. DOI: 10.1093/bioinformatics/btv730 #
#                                                                           #
#############################################################################
#                                                                           #
# BSD 3-Clause License (see https://opensource.org/licenses/BSD-3-Clause)   #
#                                                                           #
# Copyright (c) 2015-2025, Paul Macklin and the BioFVM Project              #
# All rights reserved.                                                      #
#                                                                           #
# Redistribution and use in source and binary forms, with or without        #
# modification, are permitted provided that the following conditions are    #
# met:                                                                      #
#                                                                           #
# 1. Redistributions of source code must retain the above copyright notice, #
# this list of conditions and the following disclaimer.                     #
#                                                                           #
# 2. Redistributions in binary form must reproduce the above copyright      #
# notice, this list of conditions and the following disclaimer in the       #
# documentation and/or other materials provided with the distribution.      #
#                                                                           #
# 3. Neither the name of the copyright holder nor the names of its          #
# contributors may be used to endorse or promote products derived from this #
# software without specific prior written permission.                       #
#                                                                           #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED #
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A           #
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER #
# OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,  #
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,       #
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR        #
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    #
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      #
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        #
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              #
#                                                                           #
#############################################################################
*/

#include "BioFVM_basic_agent.h"
#include "BioFVM_agent_container.h"
#include "BioFVM_vector.h" 

namespace BioFVM{

std::vector<Basic_Agent*> all_basic_agents(0); 

static int max_basic_agent_ID = 0;

void reset_max_basic_agent_ID( void )
{
	max_basic_agent_ID = 0;
}

Basic_Agent::Basic_Agent()
{
	//give the agent a unique ID  
	ID = max_basic_agent_ID; // 
	max_basic_agent_ID++; 
	// initialize position and velocity
	is_active=true;
	
	volume = 1.0; 
	
	position.assign( 3 , 0.0 ); 
	velocity.assign( 3 , 0.0 );
	previous_velocity.assign( 3 , 0.0 ); 
	// link into the microenvironment, if one is defined 
	secretion_rates= new std::vector<double>(0);
	uptake_rates= new std::vector<double>(0);
	saturation_densities= new std::vector<double>(0);
	net_export_rates = new std::vector<double>(0); 
	// extern Microenvironment* default_microenvironment;
	// register_microenvironment( default_microenvironment ); 

	internalized_substrates = new std::vector<double>(0); // 
	fraction_released_at_death = new std::vector<double>(0); 
	fraction_transferred_when_ingested = new std::vector<double>(1.0); 
	register_microenvironment( get_default_microenvironment() );
	
	// these are done in register_microenvironment
	// internalized_substrates.assign( get_default_microenvironment()->number_of_densities() , 0.0 ); 
	
	return;	
}

void Basic_Agent::update_position(double dt){ 
//make sure to update current_voxel_index if you are implementing this function
};
bool Basic_Agent::assign_position(std::vector<double> new_position)
{
	return assign_position(new_position[0], new_position[1], new_position[2]);
}

bool Basic_Agent::assign_position(double x, double y, double z)
{
	if( !get_microenvironment()->mesh.is_position_valid(x,y,z))
	{	
		// std::cout<<"Error: the new position for agent "<< ID << " is invalid: "<<x<<","<<y<<","<<"z"<<std::endl;
		return false;
	}
	position[0]=x;
	position[1]=y;
	position[2]=z;
	update_voxel_index();
	
	// make sure the agent is not already registered
	get_microenvironment()->agent_container->register_agent(this);
	return true;
}

void Basic_Agent::update_voxel_index()
{
	if( !get_microenvironment()->mesh.is_position_valid(position[0],position[1],position[2]))
	{	
		current_voxel_index=-1;
		is_active=false;
		return;
	}
	current_voxel_index= microenvironment->nearest_voxel_index( position );
}

int mycount = 0; 

void Basic_Agent::set_internal_uptake_constants( double dt )
{
	// overall form: dp/dt = S*(T-p) - U*p 
	//   p(n+1) - p(n) = dt*S(n)*T(n) - dt*( S(n) + U(n))*p(n+1)
	//   p(n+1)*temp2 =  p(n) + temp1
	//   p(n+1) = (  p(n) + temp1 )/temp2
	//int nearest_voxel= current_voxel_index;
	
/*	
	// new for tracking internal densities
	if( use_internal_densities_as_targets == true )
	{
		*saturation_densities = *internalized_substrates;
		*saturation_densities /= ( 1e-15 + volume ); 
	}
*/
	
	double internal_constant_to_discretize_the_delta_approximation = dt * volume / ( (microenvironment->voxels(current_voxel_index)).volume ) ; // needs a fix 
	
	// temp1 = dt*(V_cell/V_voxel)*S*T 
	cell_source_sink_solver_temp1.assign( (*secretion_rates).size() , 0.0 ); 
	cell_source_sink_solver_temp1 += *secretion_rates; 
	cell_source_sink_solver_temp1 *= *saturation_densities; 
	cell_source_sink_solver_temp1 *= internal_constant_to_discretize_the_delta_approximation; 
	
//	total_extracellular_substrate_change.assign( (*secretion_rates).size() , 1.0 ); 

	// temp2 = 1 + dt*(V_cell/V_voxel)*( S + U )
	cell_source_sink_solver_temp2.assign( (*secretion_rates).size() , 1.0 ); 
	axpy( &(cell_source_sink_solver_temp2) , internal_constant_to_discretize_the_delta_approximation , *secretion_rates );
	axpy( &(cell_source_sink_solver_temp2) , internal_constant_to_discretize_the_delta_approximation , *uptake_rates );	
	
	// temp for net export 
	cell_source_sink_solver_temp_export1 = *net_export_rates; 
	cell_source_sink_solver_temp_export1 *= dt; // amount exported in dt of time 
		
	cell_source_sink_solver_temp_export2 = cell_source_sink_solver_temp_export1;
	cell_source_sink_solver_temp_export2 /= ( (microenvironment->voxels(current_voxel_index)).volume ) ; 
	// change in surrounding density 
	
	volume_is_changed = false; 
	
	return; 
}

void Basic_Agent::register_microenvironment( Microenvironment* microenvironment_in )
{
	microenvironment = microenvironment_in; 	
	secretion_rates->resize( microenvironment->density_vector(0).size() , 0.0 );
	saturation_densities->resize( microenvironment->density_vector(0).size() , 0.0 );
	uptake_rates->resize( microenvironment->density_vector(0).size() , 0.0 );	
	net_export_rates->resize( microenvironment->density_vector(0).size() , 0.0 ); 

	// some solver temporary variables 
	cell_source_sink_solver_temp1.resize( microenvironment->density_vector(0).size() , 0.0 );
	cell_source_sink_solver_temp2.resize( microenvironment->density_vector(0).size() , 1.0 );
	
	cell_source_sink_solver_temp_export1.resize( microenvironment->density_vector(0).size() , 0.0 );
	cell_source_sink_solver_temp_export2.resize( microenvironment->density_vector(0).size() , 0.0 );

	// new for internalized substrate tracking 
	internalized_substrates->resize( microenvironment->density_vector(0).size() , 0.0 );
	total_extracellular_substrate_change.resize( microenvironment->density_vector(0).size() , 1.0 );
	
	fraction_released_at_death->resize( microenvironment->density_vector(0).size() , 0.0 ); 
	fraction_transferred_when_ingested->resize( microenvironment->density_vector(0).size() , 1.0 ); 

	return; 
}

void Basic_Agent::release_internalized_substrates( void )
{
	Microenvironment* pS = get_default_microenvironment();

	// A cell that has left the domain has current_voxel_index == -1 (see
	// register_microenvironment / is_out_of_domain). Indexing voxels(-1) or
	// (*pS)(-1) below is an out-of-bounds access and segfaults (notably when
	// such a cell is die()'d during an episode reset). There is no voxel to
	// release into, so skip the release for out-of-domain agents.
	if( current_voxel_index < 0 ||
	    current_voxel_index >= (int) pS->number_of_voxels() )
	{ return; }

	// change in total in voxel:
	// total_ext = total_ext + fraction*total_internal 
	// total_ext / vol_voxel = total_ext / vol_voxel + fraction*total_internal / vol_voxel 
	// density_ext += fraction * total_internal / vol_volume 
	
	*internalized_substrates /=  pS->voxels(current_voxel_index).volume; // turn to density
	*internalized_substrates *= *fraction_released_at_death;  // what fraction is released?

	// release this amount into the authoritative AoS density vector
	const unsigned int ns_r = pS->number_of_densities();
	const double* __restrict__ pi = internalized_substrates->data();
	double* __restrict__ pr = (*pS)(current_voxel_index).data();
	#pragma omp simd
	for( unsigned int s = 0; s < ns_r; s++ )
		pr[s] += pi[s];

	internalized_substrates->assign( internalized_substrates->size() , 0.0 ); 
	
	return; 
}

Microenvironment* Basic_Agent::get_microenvironment( void )
{ return microenvironment; }

Basic_Agent* create_basic_agent( void )
{
	Basic_Agent* pNew; 
	pNew = new Basic_Agent;	 
	all_basic_agents.push_back( pNew ); 
	pNew->index=all_basic_agents.size()-1;
	return pNew; 
}

void delete_basic_agent( int index )
{
	// deregister agent in microenvironment
	all_basic_agents[index]->get_microenvironment()->agent_container->remove_agent(all_basic_agents[index]);
	// de-allocate (delete) the Basic_Agent; 
	
	delete all_basic_agents[index]; 

	// next goal: remove this memory address. 

	// performance goal: don't delete in the middle -- very expensive reallocation
	// alternative: copy last element to index position, then shrink vector by 1 at the end O(constant)

	// move last item to index location  
	all_basic_agents[ all_basic_agents.size()-1 ]->index=index;
	all_basic_agents[index] = all_basic_agents[ all_basic_agents.size()-1 ];

	// shrink the vector
	all_basic_agents.pop_back();
	
	return; 
}

void delete_basic_agent( Basic_Agent* pDelete )
{
	// First, figure out the index of this agent. This is not efficient. 

	// int delete_index = 0; 
	// while( all_basic_agents[ delete_index ] != pDelete )
	// { delete_index++; }

	delete_basic_agent(pDelete->index);
	return; 
}

int Basic_Agent::get_current_voxel_index( void )
{
	return current_voxel_index;
}

std::vector<double>& Basic_Agent::nearest_density_vector( void ) 
{  
	return microenvironment->nearest_density_vector( current_voxel_index ); 
}


// directly access the gradient of substrate n nearest to the cell 
std::vector<double>& Basic_Agent::nearest_gradient( int substrate_index )
{
	return microenvironment->gradient_vector(current_voxel_index)[substrate_index]; 
}

	// directly access a vector of gradients, one gradient per substrate 
std::vector<gradient>& Basic_Agent::nearest_gradient_vector( void )
{
	return microenvironment->gradient_vector(current_voxel_index); 
}

void Basic_Agent::set_total_volume(double volume)
{
	this->volume = volume;
	volume_is_changed = true;
}

double Basic_Agent::get_total_volume()
{
	return volume;
}

const std::vector<double>& Basic_Agent::get_previous_velocity( void ) {
	return previous_velocity;
}

void Basic_Agent::simulate_secretion_and_uptake( Microenvironment* pS, double dt )
{
	if(!is_active)
	{ return; }
	
	if( volume_is_changed )
	{
		set_internal_uptake_constants(dt);
		volume_is_changed = false;
	}
	
	// Fused secretion/uptake/export, reading+writing the SoA buffer directly at this
	// cell's voxel (soa_p[s*nv + voxel], stride nv across substrates). Operating on SoA
	// lets the diffusion solver run without a per-step AoS<->SoA transpose: secretion
	// only touches occupied voxels (~cell count), far fewer than a full-field unpack.
	// AoS is synced lazily only when a non-secretion reader (sensing, I/O) needs it.
	//
	// IMPORTANT: a cell that SENSED the field earlier this step (nearest_density_vector /
	// nearest_gradient) went through the AoS accessor, which sets aos_dirty. If AoS holds
	// writes not yet in SoA, reading soa_p here would operate on stale data and silently
	// diverge from the reference (field-wide, on slow/low-diffusion substrates). Flush any
	// pending AoS writes into SoA before touching soa_p.
	pS->sync_soa_before_soa_write();
	{
		const unsigned int ns  = pS->number_of_densities();
		const unsigned int nv  = pS->number_of_voxels();
		double* __restrict__ base = pS->get_soa_p() + (unsigned int)current_voxel_index;
		const double* __restrict__ pt1  = cell_source_sink_solver_temp1.data();
		const double* __restrict__ pt2  = cell_source_sink_solver_temp2.data();
		const double* __restrict__ pex  = cell_source_sink_solver_temp_export2.data();

		if( default_microenvironment_options.track_internalized_substrates_in_each_agent == true )
		{
			const double voxel_vol = pS->voxels(current_voxel_index).volume;
			double* __restrict__ pint  = internalized_substrates->data();
			const double* __restrict__ pex1 = cell_source_sink_solver_temp_export1.data();
			for( unsigned int s = 0; s < ns; s++ )
			{
				double& rho = base[s * nv];
				const double rho_old = rho;
				const double rho_mid = (rho_old + pt1[s]) / pt2[s];
				pint[s] -= (rho_mid - rho_old) * voxel_vol;
				pint[s] -= pex1[s];
				rho = rho_mid + pex[s];
			}
		}
		else
		{
			for( unsigned int s = 0; s < ns; s++ )
			{
				double& rho = base[s * nv];
				rho = (rho + pt1[s]) / pt2[s] + pex[s];
			}
		}
	}

	return;
}

bool Basic_Agent::pack_secretion_row( double dt,
	int& voxel_out, double* temp1_out, double* temp2_out, double* export2_out )
{
	if( !is_active ) { voxel_out = -1; return false; }
	if( volume_is_changed )
	{
		set_internal_uptake_constants( dt );
		volume_is_changed = false;
	}
	voxel_out = current_voxel_index;
	const unsigned int ns = (unsigned int)cell_source_sink_solver_temp1.size();
	for( unsigned int s = 0; s < ns; s++ )
	{
		temp1_out[s]   = cell_source_sink_solver_temp1[s];
		temp2_out[s]   = cell_source_sink_solver_temp2[s];
		export2_out[s] = cell_source_sink_solver_temp_export2[s];
	}
	return true;
}

};
