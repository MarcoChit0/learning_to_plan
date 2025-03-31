; child-snack task with 1 children and 0.47 gluten factor 
; constant factor of 1.7
; random seed: 686055

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child164 - child
    bread238 - bread-portion
    content199 - content-portion
    tray486 tray301 tray462 tray346 - tray
    table318 table157 table475 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray486 kitchen)
     (at tray301 kitchen)
     (at tray462 kitchen)
     (at tray346 kitchen)
     (at_kitchen_bread bread238)
     (at_kitchen_content content199)
     (not_allergic_gluten child164)
     (waiting child164 table318)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child164)
    )
  )
)
