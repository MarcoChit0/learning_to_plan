; child-snack task with 1 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 401429

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child35 - child
    bread161 - bread-portion
    content20 - content-portion
    tray4 tray282 tray98 - tray
    table63 table225 table172 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray4 kitchen)
     (at tray282 kitchen)
     (at tray98 kitchen)
     (at_kitchen_bread bread161)
     (at_kitchen_content content20)
     (not_allergic_gluten child35)
     (waiting child35 table172)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child35)
    )
  )
)
