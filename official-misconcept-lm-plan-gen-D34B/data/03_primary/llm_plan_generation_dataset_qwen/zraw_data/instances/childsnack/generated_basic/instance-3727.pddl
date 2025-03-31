; child-snack task with 1 children and 0.45 gluten factor 
; constant factor of 1.9
; random seed: 270079

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child259 - child
    bread197 - bread-portion
    content186 - content-portion
    tray37 tray223 tray480 - tray
    table105 table160 table84 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray37 kitchen)
     (at tray223 kitchen)
     (at tray480 kitchen)
     (at_kitchen_bread bread197)
     (at_kitchen_content content186)
     (not_allergic_gluten child259)
     (waiting child259 table84)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child259)
    )
  )
)
