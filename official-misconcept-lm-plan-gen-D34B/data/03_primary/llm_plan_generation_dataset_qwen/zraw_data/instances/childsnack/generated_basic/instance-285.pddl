; child-snack task with 1 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 56744

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child395 - child
    bread173 - bread-portion
    content222 - content-portion
    tray161 - tray
    table224 table479 table229 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray161 kitchen)
     (at_kitchen_bread bread173)
     (at_kitchen_content content222)
     (not_allergic_gluten child395)
     (waiting child395 table479)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child395)
    )
  )
)
