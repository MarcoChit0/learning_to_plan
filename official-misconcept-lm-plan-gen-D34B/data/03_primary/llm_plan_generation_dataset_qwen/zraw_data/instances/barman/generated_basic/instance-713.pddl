(define (problem prob)
 (:domain barman)
 (:objects 
      shaker336 - shaker
      left right - hand
      shot177 shot309 shot242 - shot
      ingredient284 ingredient83 ingredient382 ingredient181 - ingredient
      cocktail1 - cocktail
      dispenser483 dispenser304 dispenser439 dispenser305 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker336)
  (ontable shot177)
  (ontable shot309)
  (ontable shot242)
  (dispenses dispenser483 ingredient284)
  (dispenses dispenser304 ingredient83)
  (dispenses dispenser439 ingredient382)
  (dispenses dispenser305 ingredient181)
  (clean shaker336)
  (clean shot177)
  (clean shot309)
  (clean shot242)
  (empty shaker336)
  (empty shot177)
  (empty shot309)
  (empty shot242)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker336 l0)
  (shaker-level shaker336 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient181)
  (cocktail-part2 cocktail1 ingredient83)
)
 (:goal
  (and
      (contains shot177 cocktail1)
      (contains shot309 cocktail1)
)))
