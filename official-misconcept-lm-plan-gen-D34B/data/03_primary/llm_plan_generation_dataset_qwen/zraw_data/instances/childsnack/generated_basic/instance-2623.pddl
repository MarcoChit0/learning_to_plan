; child-snack task with 1 children and 0.48 gluten factor 
; constant factor of 1.7
; random seed: 240789

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child443 - child
    bread220 - bread-portion
    content88 - content-portion
    tray243 tray254 tray68 - tray
    table153 table349 table177 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray243 kitchen)
     (at tray254 kitchen)
     (at tray68 kitchen)
     (at_kitchen_bread bread220)
     (at_kitchen_content content88)
     (not_allergic_gluten child443)
     (waiting child443 table153)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child443)
    )
  )
)
