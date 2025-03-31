; child-snack task with 1 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 27259

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child484 - child
    bread480 - bread-portion
    content357 - content-portion
    tray228 tray431 tray306 - tray
    table273 table92 table425 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray228 kitchen)
     (at tray431 kitchen)
     (at tray306 kitchen)
     (at_kitchen_bread bread480)
     (at_kitchen_content content357)
     (not_allergic_gluten child484)
     (waiting child484 table425)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child484)
    )
  )
)
