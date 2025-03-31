(define (problem prob)
 (:domain barman)
 (:objects 
      shaker47 - shaker
      left right - hand
      shot443 - shot
      ingredient341 ingredient278 - ingredient
      cocktail200 - cocktail
      dispenser62 dispenser284 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker47)
  (ontable shot443)
  (dispenses dispenser62 ingredient341)
  (dispenses dispenser284 ingredient278)
  (clean shaker47)
  (clean shot443)
  (empty shaker47)
  (empty shot443)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker47 l0)
  (shaker-level shaker47 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail200 ingredient341)
  (cocktail-part2 cocktail200 ingredient278)
)
 (:goal
  (and
      (contains shot443 cocktail200)
)))
