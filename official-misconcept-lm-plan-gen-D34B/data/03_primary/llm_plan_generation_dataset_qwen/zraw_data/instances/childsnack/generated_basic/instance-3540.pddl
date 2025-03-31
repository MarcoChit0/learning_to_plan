; child-snack task with 2 children and 0.42 gluten factor 
; constant factor of 1.3
; random seed: 162087

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child462 child406 - child
    bread285 bread151 - bread-portion
    content188 content427 - content-portion
    tray426 tray222 - tray
    table108 table183 table97 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray426 kitchen)
     (at tray222 kitchen)
     (at_kitchen_bread bread285)
     (at_kitchen_bread bread151)
     (at_kitchen_content content188)
     (at_kitchen_content content427)
     (not_allergic_gluten child406)
     (not_allergic_gluten child462)
     (waiting child462 table108)
     (waiting child406 table108)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child462)
     (served child406)
    )
  )
)
