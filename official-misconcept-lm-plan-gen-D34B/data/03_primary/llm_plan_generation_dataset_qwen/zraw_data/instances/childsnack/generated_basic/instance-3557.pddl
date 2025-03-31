; child-snack task with 2 children and 0.42 gluten factor 
; constant factor of 1.3
; random seed: 177131

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child456 child496 - child
    bread0 bread81 - bread-portion
    content459 content397 - content-portion
    tray418 tray340 - tray
    table423 table368 table217 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray418 kitchen)
     (at tray340 kitchen)
     (at_kitchen_bread bread0)
     (at_kitchen_bread bread81)
     (at_kitchen_content content459)
     (at_kitchen_content content397)
     (not_allergic_gluten child456)
     (not_allergic_gluten child496)
     (waiting child456 table423)
     (waiting child496 table368)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child456)
     (served child496)
    )
  )
)
