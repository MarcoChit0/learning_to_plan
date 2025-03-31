; child-snack task with 2 children and 0.44 gluten factor 
; constant factor of 1.9
; random seed: 76867

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child166 child244 - child
    bread341 bread400 - bread-portion
    content460 content454 - content-portion
    tray1 tray306 tray222 - tray
    table288 table43 table103 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray1 kitchen)
     (at tray306 kitchen)
     (at tray222 kitchen)
     (at_kitchen_bread bread341)
     (at_kitchen_bread bread400)
     (at_kitchen_content content460)
     (at_kitchen_content content454)
     (not_allergic_gluten child244)
     (not_allergic_gluten child166)
     (waiting child166 table288)
     (waiting child244 table103)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child166)
     (served child244)
    )
  )
)
