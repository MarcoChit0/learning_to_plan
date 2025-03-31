(define (problem prob)
 (:domain barman)
 (:objects 
      shaker251 - shaker
      left right - hand
      shot160 shot38 - shot
      ingredient312 ingredient36 ingredient420 ingredient161 - ingredient
      cocktail376 - cocktail
      dispenser189 dispenser159 dispenser261 dispenser165 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker251)
  (ontable shot160)
  (ontable shot38)
  (dispenses dispenser189 ingredient312)
  (dispenses dispenser159 ingredient36)
  (dispenses dispenser261 ingredient420)
  (dispenses dispenser165 ingredient161)
  (clean shaker251)
  (clean shot160)
  (clean shot38)
  (empty shaker251)
  (empty shot160)
  (empty shot38)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker251 l0)
  (shaker-level shaker251 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail376 ingredient312)
  (cocktail-part2 cocktail376 ingredient420)
)
 (:goal
  (and
      (contains shot160 cocktail376)
)))
