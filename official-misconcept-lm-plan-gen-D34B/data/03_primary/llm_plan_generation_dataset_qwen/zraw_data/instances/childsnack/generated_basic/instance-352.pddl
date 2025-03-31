; child-snack task with 1 children and 0.2 gluten factor 
; constant factor of 1.9
; random seed: 134000

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child438 - child
    bread193 - bread-portion
    content144 - content-portion
    tray89 tray236 tray323 tray119 - tray
    table286 table204 table183 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray89 kitchen)
     (at tray236 kitchen)
     (at tray323 kitchen)
     (at tray119 kitchen)
     (at_kitchen_bread bread193)
     (at_kitchen_content content144)
     (not_allergic_gluten child438)
     (waiting child438 table183)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child438)
    )
  )
)
