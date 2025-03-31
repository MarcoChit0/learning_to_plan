; child-snack task with 1 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 331544

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child395 - child
    bread432 - bread-portion
    content154 - content-portion
    tray85 tray397 tray168 - tray
    table429 table294 table269 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray85 kitchen)
     (at tray397 kitchen)
     (at tray168 kitchen)
     (at_kitchen_bread bread432)
     (at_kitchen_content content154)
     (not_allergic_gluten child395)
     (waiting child395 table294)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child395)
    )
  )
)
