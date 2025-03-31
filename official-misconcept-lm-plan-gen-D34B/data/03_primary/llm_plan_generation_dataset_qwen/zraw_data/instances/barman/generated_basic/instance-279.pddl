(define (problem prob)
 (:domain barman)
 (:objects 
      shaker443 - shaker
      left right - hand
      shot225 shot4 - shot
      ingredient155 ingredient154 ingredient263 - ingredient
      cocktail1 - cocktail
      dispenser63 dispenser236 dispenser434 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker443)
  (ontable shot225)
  (ontable shot4)
  (dispenses dispenser63 ingredient155)
  (dispenses dispenser236 ingredient154)
  (dispenses dispenser434 ingredient263)
  (clean shaker443)
  (clean shot225)
  (clean shot4)
  (empty shaker443)
  (empty shot225)
  (empty shot4)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker443 l0)
  (shaker-level shaker443 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient155)
  (cocktail-part2 cocktail1 ingredient263)
)
 (:goal
  (and
      (contains shot225 cocktail1)
)))
