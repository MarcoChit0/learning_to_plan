; child-snack task with 2 children and 0.54 gluten factor 
; constant factor of 1.7
; random seed: 908226

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child401 child354 - child
    bread215 bread471 - bread-portion
    content115 content58 - content-portion
    tray489 - tray
    table70 table173 table457 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray489 kitchen)
     (at_kitchen_bread bread215)
     (at_kitchen_bread bread471)
     (at_kitchen_content content115)
     (at_kitchen_content content58)
     (no_gluten_bread bread471)
     (no_gluten_content content115)
     (allergic_gluten child401)
     (not_allergic_gluten child354)
     (waiting child401 table457)
     (waiting child354 table457)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child401)
     (served child354)
    )
  )
)
