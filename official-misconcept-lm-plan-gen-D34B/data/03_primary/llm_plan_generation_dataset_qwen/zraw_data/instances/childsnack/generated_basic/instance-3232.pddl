; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.9
; random seed: 310152

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child232 child302 - child
    bread481 bread418 - bread-portion
    content78 content183 - content-portion
    tray222 tray498 tray81 tray88 - tray
    table57 table343 table463 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray222 kitchen)
     (at tray498 kitchen)
     (at tray81 kitchen)
     (at tray88 kitchen)
     (at_kitchen_bread bread481)
     (at_kitchen_bread bread418)
     (at_kitchen_content content78)
     (at_kitchen_content content183)
     (no_gluten_bread bread481)
     (no_gluten_content content78)
     (allergic_gluten child232)
     (not_allergic_gluten child302)
     (waiting child232 table463)
     (waiting child302 table343)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child232)
     (served child302)
    )
  )
)
