; child-snack task with 2 children and 0.27 gluten factor 
; constant factor of 1.7
; random seed: 777530

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child172 child300 - child
    bread50 bread60 - bread-portion
    content234 content216 - content-portion
    tray171 tray218 tray301 - tray
    table195 table162 table486 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray171 kitchen)
     (at tray218 kitchen)
     (at tray301 kitchen)
     (at_kitchen_bread bread50)
     (at_kitchen_bread bread60)
     (at_kitchen_content content234)
     (at_kitchen_content content216)
     (not_allergic_gluten child300)
     (not_allergic_gluten child172)
     (waiting child172 table486)
     (waiting child300 table486)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child172)
     (served child300)
    )
  )
)
