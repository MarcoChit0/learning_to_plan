(define (problem prob)
 (:domain barman)
 (:objects 
      shaker238 - shaker
      left right - hand
      shot163 - shot
      ingredient187 ingredient355 - ingredient
      cocktail89 - cocktail
      dispenser206 dispenser456 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker238)
  (ontable shot163)
  (dispenses dispenser206 ingredient187)
  (dispenses dispenser456 ingredient355)
  (clean shaker238)
  (clean shot163)
  (empty shaker238)
  (empty shot163)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker238 l0)
  (shaker-level shaker238 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail89 ingredient187)
  (cocktail-part2 cocktail89 ingredient355)
)
 (:goal
  (and
      (contains shot163 cocktail89)
)))
