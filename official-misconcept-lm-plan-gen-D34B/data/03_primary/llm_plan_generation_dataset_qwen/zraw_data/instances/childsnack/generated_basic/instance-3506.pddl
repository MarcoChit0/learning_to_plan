; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.7
; random seed: 297883

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child129 child391 - child
    bread359 bread218 - bread-portion
    content221 content99 - content-portion
    tray244 tray259 - tray
    table85 table20 table167 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray244 kitchen)
     (at tray259 kitchen)
     (at_kitchen_bread bread359)
     (at_kitchen_bread bread218)
     (at_kitchen_content content221)
     (at_kitchen_content content99)
     (no_gluten_bread bread218)
     (no_gluten_content content99)
     (allergic_gluten child391)
     (not_allergic_gluten child129)
     (waiting child129 table167)
     (waiting child391 table167)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child129)
     (served child391)
    )
  )
)
