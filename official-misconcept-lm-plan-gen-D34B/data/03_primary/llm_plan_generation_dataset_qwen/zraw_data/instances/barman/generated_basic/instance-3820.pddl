(define (problem prob)
 (:domain barman)
 (:objects 
      shaker428 - shaker
      left right - hand
      shot222 shot295 - shot
      ingredient39 ingredient491 ingredient419 - ingredient
      cocktail3 - cocktail
      dispenser487 dispenser241 dispenser270 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker428)
  (ontable shot222)
  (ontable shot295)
  (dispenses dispenser487 ingredient39)
  (dispenses dispenser241 ingredient491)
  (dispenses dispenser270 ingredient419)
  (clean shaker428)
  (clean shot222)
  (clean shot295)
  (empty shaker428)
  (empty shot222)
  (empty shot295)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker428 l0)
  (shaker-level shaker428 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail3 ingredient39)
  (cocktail-part2 cocktail3 ingredient491)
)
 (:goal
  (and
      (contains shot222 cocktail3)
)))
