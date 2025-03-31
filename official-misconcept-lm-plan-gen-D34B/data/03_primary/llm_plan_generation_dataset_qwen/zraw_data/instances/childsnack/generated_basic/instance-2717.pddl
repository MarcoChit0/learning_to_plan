; child-snack task with 3 children and 0.23 gluten factor 
; constant factor of 1.9
; random seed: 711558

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child35 child88 child306 - child
    bread412 bread218 bread129 - bread-portion
    content330 content469 content327 - content-portion
    tray196 tray48 tray213 - tray
    table85 table91 table307 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray196 kitchen)
     (at tray48 kitchen)
     (at tray213 kitchen)
     (at_kitchen_bread bread412)
     (at_kitchen_bread bread218)
     (at_kitchen_bread bread129)
     (at_kitchen_content content330)
     (at_kitchen_content content469)
     (at_kitchen_content content327)
     (not_allergic_gluten child88)
     (not_allergic_gluten child306)
     (not_allergic_gluten child35)
     (waiting child35 table307)
     (waiting child88 table85)
     (waiting child306 table85)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child35)
     (served child88)
     (served child306)
    )
  )
)
