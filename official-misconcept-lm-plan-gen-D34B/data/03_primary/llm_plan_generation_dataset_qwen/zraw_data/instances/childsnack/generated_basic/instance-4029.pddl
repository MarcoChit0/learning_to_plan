; child-snack task with 3 children and 0.22 gluten factor 
; constant factor of 1.9
; random seed: 653119

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child173 child352 child84 - child
    bread331 bread61 bread275 - bread-portion
    content347 content439 content483 - content-portion
    tray249 - tray
    table367 table338 table181 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray249 kitchen)
     (at_kitchen_bread bread331)
     (at_kitchen_bread bread61)
     (at_kitchen_bread bread275)
     (at_kitchen_content content347)
     (at_kitchen_content content439)
     (at_kitchen_content content483)
     (not_allergic_gluten child84)
     (not_allergic_gluten child173)
     (not_allergic_gluten child352)
     (waiting child173 table181)
     (waiting child352 table338)
     (waiting child84 table338)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child173)
     (served child352)
     (served child84)
    )
  )
)
