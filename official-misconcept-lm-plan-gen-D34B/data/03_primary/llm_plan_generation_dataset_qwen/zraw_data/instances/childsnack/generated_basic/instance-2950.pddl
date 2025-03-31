; child-snack task with 1 children and 0.69 gluten factor 
; constant factor of 1.3
; random seed: 399886

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child301 - child
    bread107 - bread-portion
    content480 - content-portion
    tray390 tray393 tray441 tray142 - tray
    table95 table417 table245 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray390 kitchen)
     (at tray393 kitchen)
     (at tray441 kitchen)
     (at tray142 kitchen)
     (at_kitchen_bread bread107)
     (at_kitchen_content content480)
     (not_allergic_gluten child301)
     (waiting child301 table95)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child301)
    )
  )
)
