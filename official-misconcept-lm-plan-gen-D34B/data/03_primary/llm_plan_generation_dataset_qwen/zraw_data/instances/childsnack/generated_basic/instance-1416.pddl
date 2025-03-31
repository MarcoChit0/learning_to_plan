; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.7
; random seed: 818622

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child332 child163 - child
    bread161 bread6 - bread-portion
    content174 content40 - content-portion
    tray291 - tray
    table434 table391 table151 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray291 kitchen)
     (at_kitchen_bread bread161)
     (at_kitchen_bread bread6)
     (at_kitchen_content content174)
     (at_kitchen_content content40)
     (no_gluten_bread bread6)
     (no_gluten_content content174)
     (allergic_gluten child163)
     (not_allergic_gluten child332)
     (waiting child332 table391)
     (waiting child163 table391)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child332)
     (served child163)
    )
  )
)
