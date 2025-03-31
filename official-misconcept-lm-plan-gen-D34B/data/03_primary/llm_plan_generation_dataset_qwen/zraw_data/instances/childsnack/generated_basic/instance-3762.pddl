; child-snack task with 1 children and 0.46 gluten factor 
; constant factor of 1.9
; random seed: 374280

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child67 - child
    bread188 - bread-portion
    content118 - content-portion
    tray434 - tray
    table187 table48 table255 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray434 kitchen)
     (at_kitchen_bread bread188)
     (at_kitchen_content content118)
     (not_allergic_gluten child67)
     (waiting child67 table187)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child67)
    )
  )
)
