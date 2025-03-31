(define (problem prob)
 (:domain barman)
 (:objects 
      shaker212 - shaker
      left right - hand
      shot386 shot367 - shot
      ingredient27 ingredient294 ingredient72 ingredient327 - ingredient
      cocktail1 - cocktail
      dispenser52 dispenser50 dispenser40 dispenser180 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker212)
  (ontable shot386)
  (ontable shot367)
  (dispenses dispenser52 ingredient27)
  (dispenses dispenser50 ingredient294)
  (dispenses dispenser40 ingredient72)
  (dispenses dispenser180 ingredient327)
  (clean shaker212)
  (clean shot386)
  (clean shot367)
  (empty shaker212)
  (empty shot386)
  (empty shot367)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker212 l0)
  (shaker-level shaker212 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient27)
  (cocktail-part2 cocktail1 ingredient72)
)
 (:goal
  (and
      (contains shot386 cocktail1)
)))
