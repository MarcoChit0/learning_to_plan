(define (problem prob)
 (:domain barman)
 (:objects 
      shaker225 - shaker
      left right - hand
      shot109 shot204 shot486 - shot
      ingredient477 ingredient173 ingredient146 ingredient123 - ingredient
      cocktail1 - cocktail
      dispenser304 dispenser50 dispenser103 dispenser48 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker225)
  (ontable shot109)
  (ontable shot204)
  (ontable shot486)
  (dispenses dispenser304 ingredient477)
  (dispenses dispenser50 ingredient173)
  (dispenses dispenser103 ingredient146)
  (dispenses dispenser48 ingredient123)
  (clean shaker225)
  (clean shot109)
  (clean shot204)
  (clean shot486)
  (empty shaker225)
  (empty shot109)
  (empty shot204)
  (empty shot486)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker225 l0)
  (shaker-level shaker225 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient173)
  (cocktail-part2 cocktail1 ingredient146)
)
 (:goal
  (and
      (contains shot109 cocktail1)
      (contains shot204 cocktail1)
)))
