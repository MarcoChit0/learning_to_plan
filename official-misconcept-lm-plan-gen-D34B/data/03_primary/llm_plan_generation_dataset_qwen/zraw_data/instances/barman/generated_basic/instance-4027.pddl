(define (problem prob)
 (:domain barman)
 (:objects 
      shaker190 - shaker
      left right - hand
      shot278 shot114 - shot
      ingredient327 ingredient137 ingredient419 - ingredient
      cocktail157 - cocktail
      dispenser479 dispenser43 dispenser33 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker190)
  (ontable shot278)
  (ontable shot114)
  (dispenses dispenser479 ingredient327)
  (dispenses dispenser43 ingredient137)
  (dispenses dispenser33 ingredient419)
  (clean shaker190)
  (clean shot278)
  (clean shot114)
  (empty shaker190)
  (empty shot278)
  (empty shot114)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker190 l0)
  (shaker-level shaker190 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail157 ingredient419)
  (cocktail-part2 cocktail157 ingredient327)
)
 (:goal
  (and
      (contains shot278 cocktail157)
)))
