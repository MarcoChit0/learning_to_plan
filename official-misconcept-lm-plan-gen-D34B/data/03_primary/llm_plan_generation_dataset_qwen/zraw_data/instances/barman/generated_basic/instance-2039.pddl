(define (problem prob)
 (:domain barman)
 (:objects 
      shaker460 - shaker
      left right - hand
      shot166 shot343 - shot
      ingredient295 ingredient278 - ingredient
      cocktail200 - cocktail
      dispenser67 dispenser484 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker460)
  (ontable shot166)
  (ontable shot343)
  (dispenses dispenser67 ingredient295)
  (dispenses dispenser484 ingredient278)
  (clean shaker460)
  (clean shot166)
  (clean shot343)
  (empty shaker460)
  (empty shot166)
  (empty shot343)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker460 l0)
  (shaker-level shaker460 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail200 ingredient295)
  (cocktail-part2 cocktail200 ingredient278)
)
 (:goal
  (and
      (contains shot166 cocktail200)
)))
