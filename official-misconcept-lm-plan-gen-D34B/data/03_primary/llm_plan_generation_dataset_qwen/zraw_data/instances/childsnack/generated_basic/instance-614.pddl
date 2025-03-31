; child-snack task with 1 children and 0.63 gluten factor 
; constant factor of 1.5
; random seed: 814995

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child251 - child
    bread152 - bread-portion
    content472 - content-portion
    tray356 - tray
    table138 table430 table248 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray356 kitchen)
     (at_kitchen_bread bread152)
     (at_kitchen_content content472)
     (not_allergic_gluten child251)
     (waiting child251 table248)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child251)
    )
  )
)
