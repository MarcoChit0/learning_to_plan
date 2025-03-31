(define (problem prob)
 (:domain barman)
 (:objects 
      shaker259 - shaker
      left right - hand
      shot374 shot406 shot443 - shot
      ingredient470 ingredient260 - ingredient
      cocktail478 - cocktail
      dispenser158 dispenser359 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker259)
  (ontable shot374)
  (ontable shot406)
  (ontable shot443)
  (dispenses dispenser158 ingredient470)
  (dispenses dispenser359 ingredient260)
  (clean shaker259)
  (clean shot374)
  (clean shot406)
  (clean shot443)
  (empty shaker259)
  (empty shot374)
  (empty shot406)
  (empty shot443)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker259 l0)
  (shaker-level shaker259 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail478 ingredient470)
  (cocktail-part2 cocktail478 ingredient260)
)
 (:goal
  (and
      (contains shot374 cocktail478)
      (contains shot406 cocktail478)
)))
