; child-snack task with 1 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 52535

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child382 - child
    bread234 - bread-portion
    content311 - content-portion
    tray8 tray82 tray53 - tray
    table492 table276 table270 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray8 kitchen)
     (at tray82 kitchen)
     (at tray53 kitchen)
     (at_kitchen_bread bread234)
     (at_kitchen_content content311)
     (not_allergic_gluten child382)
     (waiting child382 table276)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child382)
    )
  )
)
