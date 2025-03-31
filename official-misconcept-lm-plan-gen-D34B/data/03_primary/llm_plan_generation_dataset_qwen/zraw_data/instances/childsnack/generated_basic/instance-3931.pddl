; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.9
; random seed: 261892

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child51 child154 - child
    bread184 bread218 - bread-portion
    content476 content261 - content-portion
    tray195 - tray
    table136 table8 table244 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray195 kitchen)
     (at_kitchen_bread bread184)
     (at_kitchen_bread bread218)
     (at_kitchen_content content476)
     (at_kitchen_content content261)
     (no_gluten_bread bread184)
     (no_gluten_content content476)
     (allergic_gluten child51)
     (not_allergic_gluten child154)
     (waiting child51 table8)
     (waiting child154 table244)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child51)
     (served child154)
    )
  )
)
