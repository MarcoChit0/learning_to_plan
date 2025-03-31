(define (problem prob)
 (:domain barman)
 (:objects 
      shaker75 - shaker
      left right - hand
      shot434 shot264 shot430 - shot
      ingredient87 ingredient153 ingredient68 ingredient92 - ingredient
      cocktail278 - cocktail
      dispenser489 dispenser328 dispenser405 dispenser134 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker75)
  (ontable shot434)
  (ontable shot264)
  (ontable shot430)
  (dispenses dispenser489 ingredient87)
  (dispenses dispenser328 ingredient153)
  (dispenses dispenser405 ingredient68)
  (dispenses dispenser134 ingredient92)
  (clean shaker75)
  (clean shot434)
  (clean shot264)
  (clean shot430)
  (empty shaker75)
  (empty shot434)
  (empty shot264)
  (empty shot430)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker75 l0)
  (shaker-level shaker75 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail278 ingredient87)
  (cocktail-part2 cocktail278 ingredient92)
)
 (:goal
  (and
      (contains shot434 cocktail278)
      (contains shot264 ingredient68)
)))
