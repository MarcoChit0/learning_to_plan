(define (problem prob)
 (:domain barman)
 (:objects 
      shaker185 - shaker
      left right - hand
      shot368 shot136 - shot
      ingredient329 ingredient389 ingredient21 - ingredient
      cocktail416 - cocktail
      dispenser313 dispenser257 dispenser331 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker185)
  (ontable shot368)
  (ontable shot136)
  (dispenses dispenser313 ingredient329)
  (dispenses dispenser257 ingredient389)
  (dispenses dispenser331 ingredient21)
  (clean shaker185)
  (clean shot368)
  (clean shot136)
  (empty shaker185)
  (empty shot368)
  (empty shot136)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker185 l0)
  (shaker-level shaker185 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail416 ingredient329)
  (cocktail-part2 cocktail416 ingredient21)
)
 (:goal
  (and
      (contains shot368 cocktail416)
)))
