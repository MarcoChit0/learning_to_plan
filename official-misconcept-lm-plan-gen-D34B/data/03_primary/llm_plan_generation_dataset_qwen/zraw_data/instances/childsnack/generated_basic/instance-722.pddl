; child-snack task with 1 children and 0.43 gluten factor 
; constant factor of 1.9
; random seed: 98091

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child209 - child
    bread268 - bread-portion
    content136 - content-portion
    tray218 tray48 tray368 tray131 - tray
    table325 table273 table383 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray218 kitchen)
     (at tray48 kitchen)
     (at tray368 kitchen)
     (at tray131 kitchen)
     (at_kitchen_bread bread268)
     (at_kitchen_content content136)
     (not_allergic_gluten child209)
     (waiting child209 table273)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child209)
    )
  )
)
