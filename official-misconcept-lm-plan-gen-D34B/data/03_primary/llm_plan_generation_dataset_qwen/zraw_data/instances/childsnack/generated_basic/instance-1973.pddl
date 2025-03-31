; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.3
; random seed: 58544

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child332 child448 - child
    bread132 bread291 - bread-portion
    content1 content106 - content-portion
    tray342 - tray
    table328 table431 table401 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray342 kitchen)
     (at_kitchen_bread bread132)
     (at_kitchen_bread bread291)
     (at_kitchen_content content1)
     (at_kitchen_content content106)
     (no_gluten_bread bread132)
     (no_gluten_content content1)
     (allergic_gluten child332)
     (not_allergic_gluten child448)
     (waiting child332 table401)
     (waiting child448 table401)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child332)
     (served child448)
    )
  )
)
