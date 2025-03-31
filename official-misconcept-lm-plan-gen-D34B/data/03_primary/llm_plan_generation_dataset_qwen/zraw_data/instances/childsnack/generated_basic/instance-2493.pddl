; child-snack task with 2 children and 0.39 gluten factor 
; constant factor of 1.9
; random seed: 510244

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child35 child186 - child
    bread331 bread338 - bread-portion
    content17 content110 - content-portion
    tray193 tray231 - tray
    table66 table401 table419 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray193 kitchen)
     (at tray231 kitchen)
     (at_kitchen_bread bread331)
     (at_kitchen_bread bread338)
     (at_kitchen_content content17)
     (at_kitchen_content content110)
     (not_allergic_gluten child35)
     (not_allergic_gluten child186)
     (waiting child35 table66)
     (waiting child186 table401)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child35)
     (served child186)
    )
  )
)
