(define (problem prob)
 (:domain barman)
 (:objects 
      shaker417 - shaker
      left right - hand
      shot143 - shot
      ingredient164 ingredient223 ingredient156 - ingredient
      cocktail72 - cocktail
      dispenser49 dispenser409 dispenser171 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker417)
  (ontable shot143)
  (dispenses dispenser49 ingredient164)
  (dispenses dispenser409 ingredient223)
  (dispenses dispenser171 ingredient156)
  (clean shaker417)
  (clean shot143)
  (empty shaker417)
  (empty shot143)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker417 l0)
  (shaker-level shaker417 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail72 ingredient156)
  (cocktail-part2 cocktail72 ingredient223)
)
 (:goal
  (and
      (contains shot143 cocktail72)
)))
