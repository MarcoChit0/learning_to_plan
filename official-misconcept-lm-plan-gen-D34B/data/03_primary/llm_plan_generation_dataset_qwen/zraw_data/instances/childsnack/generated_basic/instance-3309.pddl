; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.3
; random seed: 282098

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child129 child344 - child
    bread226 bread370 - bread-portion
    content138 content41 - content-portion
    tray32 tray174 - tray
    table315 table184 table257 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray32 kitchen)
     (at tray174 kitchen)
     (at_kitchen_bread bread226)
     (at_kitchen_bread bread370)
     (at_kitchen_content content138)
     (at_kitchen_content content41)
     (not_allergic_gluten child129)
     (not_allergic_gluten child344)
     (waiting child129 table257)
     (waiting child344 table315)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child129)
     (served child344)
    )
  )
)
