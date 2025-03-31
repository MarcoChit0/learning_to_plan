(define (problem prob)
 (:domain barman)
 (:objects 
      shaker229 - shaker
      left right - hand
      shot361 shot439 shot195 - shot
      ingredient24 ingredient449 - ingredient
      cocktail489 - cocktail
      dispenser492 dispenser403 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker229)
  (ontable shot361)
  (ontable shot439)
  (ontable shot195)
  (dispenses dispenser492 ingredient24)
  (dispenses dispenser403 ingredient449)
  (clean shaker229)
  (clean shot361)
  (clean shot439)
  (clean shot195)
  (empty shaker229)
  (empty shot361)
  (empty shot439)
  (empty shot195)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker229 l0)
  (shaker-level shaker229 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail489 ingredient449)
  (cocktail-part2 cocktail489 ingredient24)
)
 (:goal
  (and
      (contains shot361 cocktail489)
      (contains shot439 cocktail489)
)))
