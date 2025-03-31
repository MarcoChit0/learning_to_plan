; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.9
; random seed: 388472

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child108 child302 - child
    bread106 bread116 - bread-portion
    content177 content16 - content-portion
    tray188 - tray
    table16 table425 table378 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray188 kitchen)
     (at_kitchen_bread bread106)
     (at_kitchen_bread bread116)
     (at_kitchen_content content177)
     (at_kitchen_content content16)
     (no_gluten_bread bread116)
     (no_gluten_content content16)
     (allergic_gluten child108)
     (not_allergic_gluten child302)
     (waiting child108 table16)
     (waiting child302 table425)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child108)
     (served child302)
    )
  )
)
