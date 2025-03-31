; child-snack task with 1 children and 0.69 gluten factor 
; constant factor of 1.3
; random seed: 286958

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child163 - child
    bread186 - bread-portion
    content456 - content-portion
    tray465 tray66 tray437 tray213 - tray
    table98 table340 table129 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray465 kitchen)
     (at tray66 kitchen)
     (at tray437 kitchen)
     (at tray213 kitchen)
     (at_kitchen_bread bread186)
     (at_kitchen_content content456)
     (not_allergic_gluten child163)
     (waiting child163 table340)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child163)
    )
  )
)
