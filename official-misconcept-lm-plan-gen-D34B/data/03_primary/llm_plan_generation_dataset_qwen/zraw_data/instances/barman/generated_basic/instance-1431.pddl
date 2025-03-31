(define (problem prob)
 (:domain barman)
 (:objects 
      shaker255 - shaker
      left right - hand
      shot455 shot261 - shot
      ingredient164 ingredient221 - ingredient
      cocktail366 - cocktail
      dispenser394 dispenser201 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker255)
  (ontable shot455)
  (ontable shot261)
  (dispenses dispenser394 ingredient164)
  (dispenses dispenser201 ingredient221)
  (clean shaker255)
  (clean shot455)
  (clean shot261)
  (empty shaker255)
  (empty shot455)
  (empty shot261)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker255 l0)
  (shaker-level shaker255 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail366 ingredient221)
  (cocktail-part2 cocktail366 ingredient164)
)
 (:goal
  (and
      (contains shot455 cocktail366)
)))
