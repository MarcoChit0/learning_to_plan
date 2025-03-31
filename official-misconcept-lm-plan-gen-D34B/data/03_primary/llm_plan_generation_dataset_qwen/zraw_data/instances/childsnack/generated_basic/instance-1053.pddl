; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.3
; random seed: 798016

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child490 child136 - child
    bread11 bread462 - bread-portion
    content51 content448 - content-portion
    tray461 tray269 tray415 - tray
    table18 table318 table133 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray461 kitchen)
     (at tray269 kitchen)
     (at tray415 kitchen)
     (at_kitchen_bread bread11)
     (at_kitchen_bread bread462)
     (at_kitchen_content content51)
     (at_kitchen_content content448)
     (not_allergic_gluten child136)
     (not_allergic_gluten child490)
     (waiting child490 table318)
     (waiting child136 table318)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child490)
     (served child136)
    )
  )
)
