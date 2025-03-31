(define (problem prob)
 (:domain barman)
 (:objects 
      shaker414 - shaker
      left right - hand
      shot368 shot413 shot165 - shot
      ingredient339 ingredient48 ingredient217 - ingredient
      cocktail366 - cocktail
      dispenser148 dispenser348 dispenser464 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker414)
  (ontable shot368)
  (ontable shot413)
  (ontable shot165)
  (dispenses dispenser148 ingredient339)
  (dispenses dispenser348 ingredient48)
  (dispenses dispenser464 ingredient217)
  (clean shaker414)
  (clean shot368)
  (clean shot413)
  (clean shot165)
  (empty shaker414)
  (empty shot368)
  (empty shot413)
  (empty shot165)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker414 l0)
  (shaker-level shaker414 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail366 ingredient339)
  (cocktail-part2 cocktail366 ingredient48)
)
 (:goal
  (and
      (contains shot368 cocktail366)
      (contains shot413 cocktail366)
)))
