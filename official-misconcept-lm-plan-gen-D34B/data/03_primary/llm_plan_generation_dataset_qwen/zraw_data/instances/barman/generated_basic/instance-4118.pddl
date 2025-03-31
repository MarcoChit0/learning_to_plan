(define (problem prob)
 (:domain barman)
 (:objects 
      shaker411 - shaker
      left right - hand
      shot277 shot404 - shot
      ingredient407 ingredient272 ingredient422 - ingredient
      cocktail292 - cocktail
      dispenser190 dispenser428 dispenser303 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker411)
  (ontable shot277)
  (ontable shot404)
  (dispenses dispenser190 ingredient407)
  (dispenses dispenser428 ingredient272)
  (dispenses dispenser303 ingredient422)
  (clean shaker411)
  (clean shot277)
  (clean shot404)
  (empty shaker411)
  (empty shot277)
  (empty shot404)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker411 l0)
  (shaker-level shaker411 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail292 ingredient422)
  (cocktail-part2 cocktail292 ingredient407)
)
 (:goal
  (and
      (contains shot277 cocktail292)
)))
