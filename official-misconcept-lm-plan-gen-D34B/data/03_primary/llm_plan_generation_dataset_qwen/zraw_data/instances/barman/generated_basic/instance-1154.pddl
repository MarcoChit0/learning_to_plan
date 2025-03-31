(define (problem prob)
 (:domain barman)
 (:objects 
      shaker206 - shaker
      left right - hand
      shot332 shot128 - shot
      ingredient479 ingredient331 ingredient244 ingredient310 - ingredient
      cocktail1 - cocktail
      dispenser271 dispenser42 dispenser370 dispenser241 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker206)
  (ontable shot332)
  (ontable shot128)
  (dispenses dispenser271 ingredient479)
  (dispenses dispenser42 ingredient331)
  (dispenses dispenser370 ingredient244)
  (dispenses dispenser241 ingredient310)
  (clean shaker206)
  (clean shot332)
  (clean shot128)
  (empty shaker206)
  (empty shot332)
  (empty shot128)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker206 l0)
  (shaker-level shaker206 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient244)
  (cocktail-part2 cocktail1 ingredient331)
)
 (:goal
  (and
      (contains shot332 cocktail1)
)))
