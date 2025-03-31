; child-snack task with 1 children and 0.69 gluten factor 
; constant factor of 1.3
; random seed: 141197

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child291 - child
    bread334 - bread-portion
    content480 - content-portion
    tray304 tray327 tray376 tray480 - tray
    table65 table312 table299 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray304 kitchen)
     (at tray327 kitchen)
     (at tray376 kitchen)
     (at tray480 kitchen)
     (at_kitchen_bread bread334)
     (at_kitchen_content content480)
     (not_allergic_gluten child291)
     (waiting child291 table299)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child291)
    )
  )
)
