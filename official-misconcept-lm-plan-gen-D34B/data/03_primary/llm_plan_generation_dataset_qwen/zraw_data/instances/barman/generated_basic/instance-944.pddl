(define (problem prob)
 (:domain barman)
 (:objects 
      shaker301 - shaker
      left right - hand
      shot294 shot268 shot316 - shot
      ingredient28 ingredient250 ingredient173 ingredient119 - ingredient
      cocktail1 - cocktail
      dispenser26 dispenser167 dispenser266 dispenser21 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker301)
  (ontable shot294)
  (ontable shot268)
  (ontable shot316)
  (dispenses dispenser26 ingredient28)
  (dispenses dispenser167 ingredient250)
  (dispenses dispenser266 ingredient173)
  (dispenses dispenser21 ingredient119)
  (clean shaker301)
  (clean shot294)
  (clean shot268)
  (clean shot316)
  (empty shaker301)
  (empty shot294)
  (empty shot268)
  (empty shot316)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker301 l0)
  (shaker-level shaker301 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient28)
  (cocktail-part2 cocktail1 ingredient173)
)
 (:goal
  (and
      (contains shot294 cocktail1)
      (contains shot268 cocktail1)
)))
