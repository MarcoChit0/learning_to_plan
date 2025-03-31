; child-snack task with 1 children and 0.43 gluten factor 
; constant factor of 1.9
; random seed: 978033

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child34 - child
    bread111 - bread-portion
    content209 - content-portion
    tray480 tray136 tray253 tray56 - tray
    table221 table270 table138 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray480 kitchen)
     (at tray136 kitchen)
     (at tray253 kitchen)
     (at tray56 kitchen)
     (at_kitchen_bread bread111)
     (at_kitchen_content content209)
     (not_allergic_gluten child34)
     (waiting child34 table221)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child34)
    )
  )
)
