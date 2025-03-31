(define (problem prob)
 (:domain barman)
 (:objects 
      shaker252 - shaker
      left right - hand
      shot113 shot105 shot25 - shot
      ingredient327 ingredient168 ingredient368 - ingredient
      cocktail366 - cocktail
      dispenser36 dispenser266 dispenser469 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker252)
  (ontable shot113)
  (ontable shot105)
  (ontable shot25)
  (dispenses dispenser36 ingredient327)
  (dispenses dispenser266 ingredient168)
  (dispenses dispenser469 ingredient368)
  (clean shaker252)
  (clean shot113)
  (clean shot105)
  (clean shot25)
  (empty shaker252)
  (empty shot113)
  (empty shot105)
  (empty shot25)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker252 l0)
  (shaker-level shaker252 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail366 ingredient327)
  (cocktail-part2 cocktail366 ingredient168)
)
 (:goal
  (and
      (contains shot113 cocktail366)
      (contains shot105 cocktail366)
)))
