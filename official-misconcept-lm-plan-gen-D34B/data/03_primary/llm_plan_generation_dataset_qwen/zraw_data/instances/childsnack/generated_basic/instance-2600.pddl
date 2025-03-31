; child-snack task with 1 children and 0.48 gluten factor 
; constant factor of 1.7
; random seed: 99080

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child340 - child
    bread158 - bread-portion
    content478 - content-portion
    tray251 tray350 tray25 - tray
    table64 table17 table261 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray251 kitchen)
     (at tray350 kitchen)
     (at tray25 kitchen)
     (at_kitchen_bread bread158)
     (at_kitchen_content content478)
     (not_allergic_gluten child340)
     (waiting child340 table261)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child340)
    )
  )
)
